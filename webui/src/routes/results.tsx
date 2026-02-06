import { createFileRoute } from '@tanstack/react-router'
import { createStandardSchemaV1, parseAsString, parseAsInteger } from 'nuqs'
import { SearchBar } from '@/components/search-bar'
import { MemeGrid } from '@/components/meme-grid'
import { searchMemes } from '@/lib/api'

const searchParams = {
  q: parseAsString.withDefault(''),
  limit: parseAsInteger,
  offset: parseAsInteger,
}

export const Route = createFileRoute('/results')({
  validateSearch: createStandardSchemaV1(searchParams, {
    partialOutput: true,
  }),
  loaderDeps: ({ search }) => ({ search }),
  loader: ({ deps }) => {
    if (!deps.search.q) return null
    return searchMemes({
      data: {
        q: deps.search.q,
        limit: deps.search.limit ?? undefined,
        offset: deps.search.offset ?? undefined,
      },
    })
  },
  component: ResultsPage,
})

function ResultsPage() {
  const data = Route.useLoaderData()

  return (
    <div className="min-h-screen bg-background p-4 md:p-6">
      <div className="max-w-6xl mx-auto">
        <SearchBar live />
        {data ? (
          <MemeGrid data={data} />
        ) : (
          <div className="text-center py-20 text-muted-foreground text-sm">
            <p>enter a search query above</p>
          </div>
        )}
      </div>
    </div>
  )
}
